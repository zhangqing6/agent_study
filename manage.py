#!/usr/bin/env python
"""
LangGraph Agent 项目一键管理脚本
使用: python manage.py [命令]

命令:
  start      - 一键启动所有服务 (Ollama + API)
  stop       - 停止所有服务
  restart    - 重启所有服务
  status     - 查看服务状态
  add-docs   - 添加/更新知识库文档
  list-docs  - 查看知识库中文档数量
  clean      - 清理临时文件
  help       - 显示帮助信息
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import requests
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 全局进程列表
processes = []

# 配置常量
PROJECT_ROOT = Path(__file__).parent
OLLAMA_PORT = "11435"
API_PORT = "8000"
OLLAMA_HOST = f"127.0.0.1:{OLLAMA_PORT}"
OLLAMA_BASE_URL = f"http://localhost:{OLLAMA_PORT}"
API_URL = f"http://localhost:{API_PORT}"


class ServiceManager:
    """服务管理器"""

    def __init__(self):
        self.ollama_process = None
        self.api_process = None

    def set_ollama_env(self):
        """设置 Ollama 环境变量"""
        os.environ['OLLAMA_HOST'] = OLLAMA_HOST
        logger.info(f"✅ 设置 OLLAMA_HOST={OLLAMA_HOST}")

    def check_ollama_running(self):
        """检查 Ollama 是否在运行"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def check_api_running(self):
        """检查 API 是否在运行"""
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def start_ollama(self):
        """启动 Ollama 服务"""
        if self.check_ollama_running():
            logger.info("✅ Ollama 已经在运行中")
            return True

        logger.info("🚀 正在启动 Ollama 服务...")
        self.set_ollama_env()

        try:
            # 在 Windows 上启动新窗口运行 Ollama
            self.ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                env={**os.environ, 'OLLAMA_HOST': OLLAMA_HOST},
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
            )

            # 等待 Ollama 启动
            for i in range(10):
                time.sleep(2)
                if self.check_ollama_running():
                    logger.info("✅ Ollama 启动成功")
                    return True
                logger.info(f"⏳ 等待 Ollama 启动... ({i + 1}/10)")

            logger.error("❌ Ollama 启动超时")
            return False

        except Exception as e:
            logger.error(f"❌ 启动 Ollama 失败: {e}")
            return False

    def start_api(self):
        """启动 API 服务"""
        if self.check_api_running():
            logger.info("✅ API 已经在运行中")
            return True

        logger.info("🚀 正在启动 API 服务...")

        try:
            # 创建批处理文件（临时）- 修正语法错误
            bat_content = f'''@echo off
    chcp 65001 >nul
    cd /d "{PROJECT_ROOT}"
    call "{PROJECT_ROOT}\\venv\\Scripts\\activate.bat"
    set OLLAMA_BASE_URL=http://localhost:{OLLAMA_PORT}
    echo ========================================
    echo  LangGraph Agent API 服务
    echo ========================================
    echo 启动中...
    python -m src.api.main
    pause
    '''

            bat_path = PROJECT_ROOT / "start_api.bat"
            with open(bat_path, 'w', encoding='utf-8') as f:
                f.write(bat_content)

            # 直接执行批处理文件
            os.startfile(bat_path)

            logger.info(f"✅ API 启动窗口已打开")
            logger.info(f"📝 如果窗口自动关闭，可以手动运行: {bat_path}")

            # 等待 API 启动
            for i in range(15):
                time.sleep(2)
                if self.check_api_running():
                    logger.info("✅ API 启动成功")
                    return True
                logger.info(f"⏳ 等待 API 启动... ({i + 1}/15)")

            return True

        except Exception as e:
            logger.error(f"❌ 启动 API 失败: {e}")
            return False
    def stop_ollama(self):
        """停止 Ollama 服务"""
        logger.info("🛑 正在停止 Ollama...")
        if sys.platform == "win32":
            os.system("taskkill /F /IM ollama.exe >nul 2>&1")
        else:
            os.system("pkill ollama")
        logger.info("✅ Ollama 已停止")

    def stop_api(self):
        """停止 API 服务"""
        logger.info("🛑 正在停止 API...")
        if sys.platform == "win32":
            os.system("taskkill /F /IM python.exe /FI \"WINDOWTITLE eq *src.api.main*\" >nul 2>&1")
        logger.info("✅ API 已停止")

    def show_status(self):
        """显示所有服务状态"""
        logger.info("=" * 50)
        logger.info("📊 服务状态检查")
        logger.info("=" * 50)

        # 检查 Ollama
        if self.check_ollama_running():
            logger.info("✅ Ollama: 运行中")
            try:
                models = requests.get(f"{OLLAMA_BASE_URL}/api/tags").json()
                logger.info(f"   └─ 可用模型: {', '.join([m['name'] for m in models.get('models', [])])}")
            except:
                pass
        else:
            logger.info("❌ Ollama: 未运行")

        # 检查 API
        if self.check_api_running():
            logger.info("✅ API: 运行中")
            logger.info(f"   └─ 文档地址: http://localhost:{API_PORT}/docs")
        else:
            logger.info("❌ API: 未运行")

        # 检查 Docker 服务
        try:
            result = subprocess.run(["docker-compose", "ps"], capture_output=True, text=True)
            if "Up" in result.stdout:
                logger.info("✅ Docker 服务: 运行中")
                for line in result.stdout.split('\n'):
                    if "Up" in line:
                        logger.info(f"   └─ {line.strip()}")
            else:
                logger.info("⚠️ Docker 服务: 未完全启动")
        except:
            logger.info("⚠️ Docker 服务: 无法检查")

    def start_all(self):
        """一键启动所有服务"""
        logger.info("=" * 50)
        logger.info("🚀 一键启动所有服务")
        logger.info("=" * 50)

        # 1. 检查 Docker 服务
        logger.info("📦 检查 Docker 服务...")
        try:
            subprocess.run(["docker-compose", "ps"], check=True, capture_output=True)
            logger.info("✅ Docker 服务已就绪")
        except:
            logger.warning("⚠️ Docker 服务未启动，尝试启动...")
            os.system("docker-compose up -d")
            time.sleep(5)

        # 2. 启动 Ollama
        if not self.start_ollama():
            logger.error("❌ Ollama 启动失败，终止启动流程")
            return False

        # 3. 启动 API
        if not self.start_api():
            logger.error("❌ API 启动失败")
            return False

        logger.info("=" * 50)
        logger.info("✅ 所有服务启动完成！")
        logger.info(f"📝 API 文档: http://localhost:{API_PORT}/docs")
        logger.info(f"🔍 健康检查: http://localhost:{API_PORT}/health")
        logger.info("=" * 50)
        return True

    def add_documents(self):
        """添加文档到知识库"""
        logger.info("=" * 50)
        logger.info("📚 添加文档到知识库")
        logger.info("=" * 50)

        # 检查 API 是否运行
        if not self.check_api_running():
            logger.warning("⚠️ API 未运行，文档添加后需要启动 API 才能查询")

        # 检查数据目录
        data_dir = PROJECT_ROOT / "src" / "data" / "raw"
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            logger.info(f"📁 创建数据目录: {data_dir}")

        # 列出当前可用的文档
        files = list(data_dir.glob("*.*"))
        if files:
            logger.info("📄 当前待处理文档:")
            for f in files:
                logger.info(f"   - {f.name} ({f.stat().st_size / 1024:.1f}KB)")
        else:
            logger.info("📂 数据目录为空，请将文档放入:")
            logger.info(f"   {data_dir}")

        # 运行添加脚本
        logger.info("⏳ 正在处理文档...")
        venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
        result = subprocess.run([str(venv_python), "add_docs.py"], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ 文档添加成功！")
            # 提取添加的文档块数量
            for line in result.stdout.split('\n'):
                if "总共添加" in line:
                    logger.info(f"📊 {line.strip()}")
        else:
            logger.error("❌ 文档添加失败")
            logger.error(result.stderr)

    def list_knowledge_base(self):
        """查看知识库状态"""
        logger.info("=" * 50)
        logger.info("📊 知识库状态")
        logger.info("=" * 50)

        try:
            # 使用 Python 脚本查询 Milvus
            venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
            script = """
from pymilvus import connections, Collection
import sys
try:
    connections.connect(host='localhost', port='19530')
    collection = Collection('knowledge_base')
    collection.load()
    count = collection.num_entities
    print(f"知识库中文档块数量: {count}")
except Exception as e:
    print(f"查询失败: {e}")
    sys.exit(1)
"""
            result = subprocess.run(
                [str(venv_python), "-c", script],
                capture_output=True, text=True
            )
            logger.info(result.stdout.strip())

        except Exception as e:
            logger.error(f"查询失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='LangGraph Agent 项目管理工具')
    parser.add_argument('command', nargs='?', default='help',
                        choices=['start', 'stop', 'restart', 'status',
                                 'add-docs', 'list-docs', 'help'],
                        help='要执行的命令')

    args = parser.parse_args()
    manager = ServiceManager()

    if args.command == 'start':
        manager.start_all()

    elif args.command == 'stop':
        logger.info("🛑 停止所有服务...")
        manager.stop_api()
        manager.stop_ollama()
        logger.info("✅ 所有服务已停止")

    elif args.command == 'restart':
        logger.info("🔄 重启所有服务...")
        manager.stop_api()
        manager.stop_ollama()
        time.sleep(3)
        manager.start_all()

    elif args.command == 'status':
        manager.show_status()

    elif args.command == 'add-docs':
        manager.add_documents()

    elif args.command == 'list-docs':
        manager.list_knowledge_base()

    else:  # help
        print(__doc__)


if __name__ == "__main__":
    main()