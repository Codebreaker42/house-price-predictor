FROM python:3.10-slim

WORKDIR /streamlit_app

COPY streamlit_app/ /streamlit_app/

# COPY streamlit_app/df.pkl /docker_app/df.pkl

# Show pip logs in real-time
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --progress-bar=force -r requirements.txt

EXPOSE 5000

# local 
CMD [ "python" , "app.py" ]

#Prod
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]