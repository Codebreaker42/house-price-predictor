FROM python:3.9 

WORKDIR /docker_app

COPY streamlit_app/ /docker_app/

# COPY streamlit_app/df.pkl /docker_app/df.pkl

RUN pip install -r requirements.txt 

EXPOSE 5000

# local 
CMD [ "python" , "app.py" ]

#Prod
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]