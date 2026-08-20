FROM python:3.10-slim

# tesseract-ocr is a system package (not pip-installable) needed by the
# "import team from photo" feature (pytesseract just calls out to this
# binary). Render's native Python environment can't apt-get install
# anything, which is the whole reason this app is deployed via Docker
# instead of Render's buildpack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installed as its own layer so `docker build` only re-runs pip install
# when a requirements file actually changes, not on every code edit.
COPY requirements.txt ./requirements-root.txt
COPY webapp/requirements.txt ./requirements-webapp.txt
RUN pip install --no-cache-dir -r requirements-root.txt -r requirements-webapp.txt

COPY . .

# Render sets $PORT itself and routes public traffic to whatever port the
# app actually binds -- webapp/app.py already reads it via
# os.environ.get("PORT", 8888), so nothing else is needed here.
CMD ["python3", "webapp/app.py"]
