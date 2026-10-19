# הבוט של אתר הפניקס - אימג' אחד שרץ בכל ענן שמריץ קונטיינרים (AWS App Runner / ECS, Hugging Face Spaces,
# Google Cloud Run, Render). המפתחות לא באימג': משתני סביבה (Secrets של הענן) - בדיוק השמות שב-.env.example.
#
# בנייה:  docker build -t phoenix-bot .
# הרצה:   docker run --env-file .env -p 8501:8501 phoenix-bot     -> http://localhost:8501
#
# האינדקס (data/chroma) נכנס לאימג' כמו שהוא - לא בונים אותו מחדש בענן. המדרג המקומי (גיבוי ל-Cohere) מורד
# בזמן הבנייה, כדי שלא יורד בכל הפעלה.
# 3.12 ולא 3.13: ב-3.13 אימות התעודות מחמיר כברירת מחדל ונכשל מאחורי סינון HTTPS (Netspark). בענן אין הבדל.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
# רשת עם סינון HTTPS (למשל Netspark): תעודת השורש שלה ב-certs/*.crt (לא ב-Git) נכנסת למאגר התעודות - בלי לכבות
# אימות. בענן התיקייה ריקה (רק README), ושום דבר לא משתנה.
COPY certs/ /tmp/certs/
RUN cp /tmp/certs/*.crt /usr/local/share/ca-certificates/ 2>/dev/null; update-ca-certificates >/dev/null
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN test -f data/chroma/chroma.sqlite3 || (echo "חסר האינדקס data/chroma - ראו README" && exit 1)
# המדרג המקומי: בבנייה מקומית הוא כבר ב-data/models (מועתק) ומדלגים; בענן - מורד כאן, פעם אחת
RUN python -m phoenix.rerank download

# פורט: הענן קובע ב-PORT (Cloud Run, Render, App Runner); ברירת מחדל 8501
ENV PORT=8501
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/_stcore/health' % os.environ['PORT'])"
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]
