FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# 先装依赖以利用层缓存。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码与验收测试都进入镜像，使 verify 服务可一次性运行。
COPY app ./app
COPY tests ./tests
COPY pyproject.toml ./

EXPOSE 8000

# 默认启动 API；verify 服务在 compose 中覆盖为 pytest。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
