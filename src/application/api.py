import os
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from .decision_system import DecisionSupportSystem
from ..utils.logger import get_logger

logger = get_logger(__name__)


# 定义请求和响应模型
class RecommendationRequest(BaseModel):
    scenario: Optional[str] = "normal"
    version_info: Optional[Dict[str, Any]] = None
    holidays: Optional[List[int]] = None


class RecommendationResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class APIHandler:
    """API处理器，提供RESTful接口"""

    def __init__(self, config_path: str = "config/application.yaml"):
        """初始化API处理器

        Args:
            config_path: 应用配置文件路径
        """
        self.config = self._load_config(config_path)
        self.api_config = self.config['application']['api']

        # 初始化决策支持系统
        self.dss = None
        if self.api_config['enabled']:
            self.dss = DecisionSupportSystem(config_path)

        # 创建FastAPI应用
        self.app = FastAPI(
            title="手机发布版本编排API",
            description="提供版本发布决策支持的RESTful API",
            version="1.0.0"
        )

        # 配置CORS
        if self.api_config['cors_enabled']:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # 注册路由
        self._setup_routes()

        logger.info("API handler initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_routes(self) -> None:
        """设置API路由"""

        @self.app.get("/")
        async def root():
            return {"message": "手机发布版本编排API", "version": "1.0.0"}

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "mobile_release_api"}

        @self.app.post("/recommendations", response_model=RecommendationResponse)
        async def get_recommendation(request: RecommendationRequest):
            """获取发布推荐"""
            try:
                if not self.dss:
                    raise HTTPException(status_code=503, detail="Service not available")

                # 准备状态信息
                state_info = {}
                if request.version_info:
                    state_info['version_info'] = request.version_info
                if request.holidays:
                    state_info['holidays'] = request.holidays

                # 生成推荐
                recommendation = self.dss.generate_recommendation(
                    state_info, request.scenario
                )

                return RecommendationResponse(
                    success=True,
                    message="Recommendation generated successfully",
                    data=recommendation
                )

            except Exception as e:
                logger.error(f"Failed to generate recommendation: {e}")
                return RecommendationResponse(
                    success=False,
                    message="Failed to generate recommendation",
                    error=str(e)
                )

        @self.app.get("/recommendations/{recommendation_id}")
        async def get_recommendation_by_id(recommendation_id: str):
            """根据ID获取推荐结果"""
            try:
                # 查找推荐文件
                file_path = os.path.join("results", "recommendations", f"{recommendation_id}.json")

                if not os.path.exists(file_path):
                    raise HTTPException(status_code=404, detail="Recommendation not found")

                # 返回文件内容
                return FileResponse(file_path, media_type="application/json")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get recommendation: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/scenarios")
        async def get_available_scenarios():
            """获取可用场景列表"""
            try:
                scenarios_dir = "config/scenarios"
                scenarios = []

                if os.path.exists(scenarios_dir):
                    for file in os.listdir(scenarios_dir):
                        if file.endswith(".yaml") or file.endswith(".yml"):
                            scenario_name = file.replace("_scenario.yaml", "").replace("_scenario.yml", "")
                            scenarios.append(scenario_name)

                return {"scenarios": scenarios}

            except Exception as e:
                logger.error(f"Failed to get scenarios: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/models")
        async def get_available_models():
            """获取可用模型列表"""
            try:
                models_dir = "models"
                models = {"best": [], "checkpoints": []}

                # 获取最佳模型
                best_dir = os.path.join(models_dir, "best")
                if os.path.exists(best_dir):
                    for file in os.listdir(best_dir):
                        if file.endswith(".zip"):
                            models["best"].append(file)

                # 获取检查点模型
                checkpoints_dir = os.path.join(models_dir, "checkpoints")
                if os.path.exists(checkpoints_dir):
                    for file in os.listdir(checkpoints_dir):
                        if file.endswith(".zip"):
                            models["checkpoints"].append(file)

                return models

            except Exception as e:
                logger.error(f"Failed to get models: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def run(self) -> None:
        """运行API服务器"""
        if not self.api_config['enabled']:
            logger.warning("API is disabled in configuration")
            return

        host = self.api_config['host']
        port = self.api_config['port']
        debug = self.api_config['debug']

        logger.info(f"Starting API server on {host}:{port}")

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            debug=debug,
            log_level="info" if debug else "warning"
        )

    def close(self) -> None:
        """关闭API处理器"""
        if self.dss:
            self.dss.close()
        logger.info("API handler closed")