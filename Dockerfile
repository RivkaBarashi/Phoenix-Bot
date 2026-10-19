# הבוט של אתר הפניקס - אימג' אחד שרץ בכל ענן שמריץ קונטיינרים (Render, AWS App Runner / ECS, Google Cloud Run).
# המפתחות לא באימג': משתני סביבה (Secrets של הענן) - בדיוק השמות שב-.env.example.
#
# בנייה:  docker build -t phoenix-bot .
# הרצה:   docker run --env-file .env -p 8501:8501 phoenix-bot     -> http://localhost:8501
#
# האינדקס (data/chroma, ב-Git) נכנס לאימג' כמו שהוא - לא בונים אותו מחדש בענן.
# המדרג המקומי לא נכנס כברירת מחדל: על מעבד של ענן חינמי הוא ~10 שניות לשאלה (לא נכנס ב-20 השניות) ו-1.5GB
# זיכרון (ב-Render החינמי יש 512MB). הגיבוי ל-Cohere בענן הוא Voyage. בשרת חזק: --build-arg WITH_LOCAL_RERANKER=1.
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
ARG WITH_LOCAL_RERANKER=0
RUN if [ "$WITH_LOCAL_RERANKER" = "1" ]; then python -m phoenix.rerank download; fi

# פורט: הענן קובע ב-PORT (Cloud Run, Render, App Runner); ברירת מחדל 8501
ENV PORT=8501
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/_stcore/health' % os.environ['PORT'])"
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]
