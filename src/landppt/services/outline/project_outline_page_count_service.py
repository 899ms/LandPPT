import asyncio
import base64
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...api.models import (
    PPTGenerationRequest,
    PPTOutline,
    EnhancedPPTOutline,
    SlideContent,
    PPTProject,
    TodoBoard,
    FileOutlineGenerationResponse,
)
from ...ai import get_ai_provider, get_role_provider, AIMessage, MessageRole
from ...ai.base import TextContent, ImageContent
from ...core.config import ai_config, app_config
from ..runtime.ai_execution import ExecutionContext, scoped_ai_conversation
from ..prompts import prompts_manager
from ..research.enhanced_research_service import EnhancedResearchService
from ..research.enhanced_report_generator import EnhancedReportGenerator
from ..pyppeteer_pdf_converter import get_pdf_converter
from ..image.image_service import ImageService
from ..image.adapters.ppt_prompt_adapter import PPTSlideContext
from ...utils.thread_pool import run_blocking_io, to_thread


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .project_outline_generation_service import ProjectOutlineGenerationService

class ProjectOutlinePageCountService:
    """Extracted logic from ProjectOutlineGenerationService."""

    def __init__(self, service: 'ProjectOutlineGenerationService'):
        self._service = service

    def __getattr__(self, name: str):
        return getattr(self._service, name)

    async def _mark_outline_generation_failed(self, project_id: str, reason: str) -> None:
        """Record a real failure on the stage so the UI can offer a retry."""
        try:
            from ..db_project_manager import DatabaseProjectManager
            db_manager = DatabaseProjectManager()
            await db_manager.update_stage_status(
                project_id,
                'outline_generation',
                'failed',
                None,
                {'message': reason, 'failed_at': time.time()},
            )
            logger.info('Marked outline_generation as failed for project %s', project_id)
        except Exception as stage_error:
            logger.error(
                'Could not mark outline_generation as failed for project %s: %s',
                project_id, stage_error
            )

    @scoped_ai_conversation("outline")
    async def _execute_outline_generation(self, project_id: str, confirmed_requirements: Dict[str, Any], system_prompt: str) -> str:
        """Execute outline generation as a complete task"""
        try:
            project = await self.project_manager.get_project(project_id)
            existing_outline = project.outline if project and isinstance(project.outline, dict) else None
            existing_slides = existing_outline.get('slides', []) if existing_outline else []
            if existing_slides:
                logger.info('Project %s already has outline with %s slides, reusing existing outline', project_id, len(existing_slides))
                try:
                    await self._update_outline_generation_stage(project_id, existing_outline)
                except Exception as stage_error:
                    logger.warning('Failed to mark reused outline generation stage as completed for project %s: %s', project_id, stage_error)
                return f"✅ PPT大纲已存在，跳过重复生成。\n\n标题：{existing_outline.get('title', confirmed_requirements.get('topic', '未知'))}\n页数：{len(existing_slides)}页\n已复用现有大纲"

            page_count_settings = confirmed_requirements.get('page_count_settings', {})
            page_count_mode = page_count_settings.get('mode', 'ai_decide')
            page_count_instruction = ''
            expected_page_count = None
            if page_count_mode == 'custom_range':
                min_pages = page_count_settings.get('min_pages') or 8
                max_pages = page_count_settings.get('max_pages') or 15
                page_count_instruction = f'- 页数要求：必须严格生成{min_pages}-{max_pages}页的PPT。请确保生成的幻灯片数量在此范围内，不能超出或不足。'
                expected_page_count = {'min': min_pages, 'max': max_pages, 'mode': 'range'}
                logger.info(f'Custom page count range set: {min_pages}-{max_pages} pages')
            elif page_count_mode == 'fixed':
                # 'fixed' used to fall into the else branch below, so the prompt told
                # the model to pick its own page count and the user's setting was
                # then overwritten with {'mode': 'ai_decide'} in the metadata.
                fixed_pages = page_count_settings.get('fixed_pages') or 10
                page_count_instruction = (
                    f'- 页数要求：必须严格生成恰好{fixed_pages}页的PPT，'
                    f'不能多于或少于{fixed_pages}页。'
                )
                expected_page_count = {
                    'min': fixed_pages,
                    'max': fixed_pages,
                    'fixed_pages': fixed_pages,
                    'mode': 'fixed',
                }
                logger.info(f'Fixed page count set: {fixed_pages} pages')
            else:
                page_count_instruction = '- 页数要求：请根据主题内容的复杂度、深度和逻辑结构，自主决定最合适的页数，确保内容充实且逻辑清晰'
                expected_page_count = {'mode': 'ai_decide'}
                logger.info('AI decide mode set for page count')
            topic = confirmed_requirements['topic']
            target_audience = confirmed_requirements.get('target_audience', '普通大众')
            ppt_style = confirmed_requirements.get('ppt_style', 'general')
            custom_style = confirmed_requirements.get('custom_style_prompt', '无')
            description = confirmed_requirements.get('description', '无')
            context = prompts_manager.get_outline_generation_context(topic=topic, target_audience=target_audience, page_count_instruction=page_count_instruction, ppt_style=ppt_style, custom_style=custom_style, description=description, page_count_mode=page_count_mode)
            response = await self._text_completion_for_role('outline', prompt=context, system_prompt=system_prompt, temperature=ai_config.temperature)
            import json
            import re
            try:
                content = response.content.strip()
                # Use the shared candidate parser instead of a bespoke regex. The old
                # fallback pattern only matched braces nested <= 2 deep, so an outline
                # containing e.g. "chart_config": {...} made the root object
                # unmatchable and the regex captured a single inner slide instead,
                # losing the whole outline.
                outline_data = self._parse_json_like_outline(content)
                if outline_data is None:
                    raise ValueError('模型响应中未找到可解析的JSON大纲')
                outline_data = self._standardize_outline_format(outline_data)
                outline_data = await self._validate_and_repair_outline_json(outline_data, confirmed_requirements)
                if expected_page_count and 'slides' in outline_data:
                    actual_page_count = len(outline_data['slides'])
                    logger.info(f'Generated outline has {actual_page_count} pages')
                    # 'fixed' is enforced here too: it is just a range whose bounds
                    # are equal, and it previously got no enforcement at all.
                    if expected_page_count['mode'] in ('range', 'fixed'):
                        min_pages = expected_page_count['min']
                        max_pages = expected_page_count['max']
                        if actual_page_count < min_pages or actual_page_count > max_pages:
                            logger.warning(f'Generated outline has {actual_page_count} pages, but expected {min_pages}-{max_pages} pages. Adjusting...')
                            outline_data = await self._adjust_outline_page_count(outline_data, min_pages, max_pages, confirmed_requirements)
                            adjusted_page_count = len(outline_data.get('slides', []))
                            logger.info(f'Adjusted outline to {adjusted_page_count} pages')
                            if adjusted_page_count < min_pages or adjusted_page_count > max_pages:
                                logger.error(f'Failed to adjust page count to required range {min_pages}-{max_pages}')
                                target_pages = (min_pages + max_pages) // 2
                                outline_data = await self._force_page_count(outline_data, target_pages, confirmed_requirements)
                        else:
                            logger.info(f'Page count {actual_page_count} is within required range {min_pages}-{max_pages}')
                    if 'metadata' not in outline_data:
                        outline_data['metadata'] = {}
                    # Record what the USER asked for; the resolved bounds go alongside
                    # it. Writing expected_page_count over the top used to report
                    # mode='ai_decide' for a project that requested a fixed count.
                    outline_data['metadata']['page_count_settings'] = dict(page_count_settings) or expected_page_count
                    outline_data['metadata']['resolved_page_count'] = expected_page_count
                    outline_data['metadata']['actual_page_count'] = len(outline_data.get('slides', []))
                project = await self.project_manager.get_project(project_id)
                if project:
                    project.outline = outline_data
                    project.updated_at = time.time()
                    logger.info(f'Successfully saved outline to memory for project {project_id}')
                try:
                    from ..db_project_manager import DatabaseProjectManager
                    db_manager = DatabaseProjectManager()
                    save_success = await db_manager.save_project_outline(project_id, outline_data)
                    if save_success:
                        logger.info(f'✅ Successfully saved outline to database for project {project_id}')
                        saved_project = await db_manager.get_project(project_id)
                        if saved_project and saved_project.outline:
                            saved_slides_count = len(saved_project.outline.get('slides', []))
                            logger.info(f'✅ Verified: outline saved with {saved_slides_count} slides')
                        else:
                            logger.error(f'❌ Verification failed: outline not found in database')
                            return f'❌ 大纲保存失败：数据库验证失败'
                    else:
                        logger.error(f'❌ Failed to save outline to database for project {project_id}')
                        return f'❌ 大纲保存失败：数据库写入失败'
                except Exception as save_error:
                    logger.error(f'❌ Exception while saving outline to database: {save_error}')
                    import traceback
                    traceback.print_exc()
                    return f'❌ 大纲保存失败：{str(save_error)}'
                try:
                    from ..db_project_manager import DatabaseProjectManager
                    db_manager = DatabaseProjectManager()
                    await db_manager.update_stage_status(project_id, 'outline_generation', 'completed', 100.0, {'outline_title': outline_data.get('title', '未知'), 'slides_count': len(outline_data.get('slides', [])), 'completed_at': time.time()})
                    logger.info(f'Successfully updated outline generation stage to completed for project {project_id}')
                except Exception as stage_error:
                    logger.error(f'Failed to update outline generation stage status: {stage_error}')
                final_page_count = len(outline_data.get('slides', []))
                return f"✅ PPT大纲生成完成！\n\n标题：{outline_data.get('title', '未知')}\n页数：{final_page_count}页\n已保存到数据库\n\n{response.content}"
            except Exception as e:
                logger.error(f'Error parsing outline JSON: {e}')
                logger.error(f'Response content: {response.content[:500]}...')
                # Previously a generic 3-page placeholder was saved here and reported
                # with a success marker, so a failed generation looked successful and
                # silently replaced the user's topic with filler content.
                # Fail loudly instead, and keep any existing outline untouched.
                await self._mark_outline_generation_failed(project_id, str(e))
                return (
                    f'❌ 大纲生成失败：模型返回的内容无法解析为有效大纲（{str(e)}）。\n'
                    f'请重试，或调整主题描述后重新生成。'
                )
        except Exception as e:
            logger.error(f'Error in outline generation: {e}')
            raise

    async def _adjust_outline_page_count(self, outline_data: Dict[str, Any], min_pages: int, max_pages: int, confirmed_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust outline page count to meet requirements"""
        try:
            current_slides = outline_data.get('slides', [])
            current_count = len(current_slides)
            if current_count < min_pages:
                logger.info(f'Adding slides to meet minimum requirement: {current_count} -> {min_pages}')
                outline_data = await self._expand_outline(outline_data, min_pages, confirmed_requirements)
            elif current_count > max_pages:
                logger.info(f'Reducing slides to meet maximum requirement: {current_count} -> {max_pages}')
                outline_data = await self._condense_outline(outline_data, max_pages)
            return outline_data
        except Exception as e:
            logger.error(f'Error adjusting outline page count: {e}')
            return outline_data

    async def _expand_outline(self, outline_data: Dict[str, Any], target_pages: int, confirmed_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Expand outline to reach target page count"""
        try:
            slides = outline_data.get('slides', [])
            current_count = len(slides)
            needed_slides = target_pages - current_count
            topic = confirmed_requirements.get('topic', outline_data.get('title', ''))
            focus_content = confirmed_requirements.get('focus_content', [])
            conclusion_slide = None
            if slides and slides[-1].get('slide_type') in ['thankyou', 'conclusion']:
                conclusion_slide = slides.pop()
            for i in range(needed_slides):
                page_number = len(slides) + 1
                if i < len(focus_content):
                    new_slide = {'page_number': page_number, 'title': focus_content[i], 'content_points': [f'{focus_content[i]}的详细介绍', '核心要点', '实际应用'], 'slide_type': 'content', 'description': f'详细介绍{focus_content[i]}相关内容'}
                else:
                    new_slide = {'page_number': page_number, 'title': f'{topic} - 补充内容 {i + 1}', 'content_points': ['补充要点1', '补充要点2', '补充要点3'], 'slide_type': 'content', 'description': f'关于{topic}的补充内容'}
                slides.append(new_slide)
            if conclusion_slide:
                conclusion_slide['page_number'] = len(slides) + 1
                slides.append(conclusion_slide)
            for i, slide in enumerate(slides):
                slide['page_number'] = i + 1
            outline_data['slides'] = slides
            return outline_data
        except Exception as e:
            logger.error(f'Error expanding outline: {e}')
            return outline_data

    async def _condense_outline(self, outline_data: Dict[str, Any], target_pages: int) -> Dict[str, Any]:
        """Condense outline to reach target page count"""
        try:
            slides = outline_data.get('slides', [])
            current_count = len(slides)
            if current_count <= target_pages:
                return outline_data
            title_slides = [s for s in slides if s.get('slide_type') in ['title', 'cover']]
            conclusion_slides = [s for s in slides if s.get('slide_type') in ['thankyou', 'conclusion']]
            content_slides = [s for s in slides if s.get('slide_type') not in ['title', 'cover', 'thankyou', 'conclusion']]
            # When the reserved title/closing pages alone meet or exceed the target,
            # the previous code left ALL content slides in place and returned more
            # pages than the maximum. Trim the reserved pages instead so the target
            # is actually reachable.
            title_slides = title_slides[:1]
            available_content_slots = target_pages - (len(title_slides) + len(conclusion_slides))
            if available_content_slots <= 0:
                conclusion_slides = []
                available_content_slots = target_pages - len(title_slides)
            if available_content_slots <= 0:
                title_slides = []
                available_content_slots = target_pages
            if len(content_slides) > available_content_slots:
                content_slides = content_slides[:max(available_content_slots, 0)]
            new_slides = title_slides + content_slides + conclusion_slides
            for i, slide in enumerate(new_slides):
                slide['page_number'] = i + 1
            outline_data['slides'] = new_slides
            return outline_data
        except Exception as e:
            logger.error(f'Error condensing outline: {e}')
            return outline_data

    async def _force_page_count(self, outline_data: Dict[str, Any], target_pages: int, confirmed_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Force outline to exact page count"""
        try:
            slides = outline_data.get('slides', [])
            current_count = len(slides)
            logger.info(f'Forcing page count from {current_count} to {target_pages}')
            if current_count == target_pages:
                return outline_data
            title_slides = [s for s in slides if s.get('slide_type') in ['title', 'cover']]
            conclusion_slides = [s for s in slides if s.get('slide_type') in ['thankyou', 'conclusion']]
            content_slides = [s for s in slides if s.get('slide_type') not in ['title', 'cover', 'thankyou', 'conclusion']]
            # Keep at most one title and one closing page so the content budget is
            # never negative. Previously a target smaller than the reserved page
            # count collapsed the entire deck to a single page (or zero), silently
            # violating the user's minimum.
            title_slides = title_slides[:1]
            conclusion_slides = conclusion_slides[:1]
            content_slots_needed = target_pages - (len(title_slides) + len(conclusion_slides))
            if content_slots_needed < 0:
                conclusion_slides = []
                content_slots_needed = target_pages - len(title_slides)
            if content_slots_needed < 0:
                title_slides = []
                content_slots_needed = target_pages

            if content_slots_needed == 0:
                new_slides = (title_slides + conclusion_slides)[:target_pages]
            else:
                if len(content_slides) > content_slots_needed:
                    content_slides = content_slides[:content_slots_needed]
                elif len(content_slides) < content_slots_needed:
                    topic = confirmed_requirements.get('topic', outline_data.get('title', ''))
                    focus_content = confirmed_requirements.get('focus_content', [])
                    for i in range(content_slots_needed - len(content_slides)):
                        page_number = len(content_slides) + i + 1
                        if i < len(focus_content):
                            new_slide = {'page_number': page_number, 'title': focus_content[i], 'content_points': [f'{focus_content[i]}的详细介绍', '核心要点', '实际应用'], 'slide_type': 'content', 'description': f'详细介绍{focus_content[i]}相关内容'}
                        else:
                            new_slide = {'page_number': page_number, 'title': f'{topic} - 内容 {i + 1}', 'content_points': ['要点1', '要点2', '要点3'], 'slide_type': 'content', 'description': f'关于{topic}的内容'}
                        content_slides.append(new_slide)
                new_slides = title_slides + content_slides + conclusion_slides
            for i, slide in enumerate(new_slides):
                slide['page_number'] = i + 1
            outline_data['slides'] = new_slides
            logger.info(f'Successfully forced page count to {len(new_slides)} pages')
            return outline_data
        except Exception as e:
            logger.error(f'Error forcing page count: {e}')
            return outline_data
