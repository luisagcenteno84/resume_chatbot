FROM python:3.10-slim

COPY . /src
COPY public /public
COPY chainlit.md /

# 👇
ARG GIT_HASH
ENV GOOGLE_API_KEY=AIzaSyCeN8OmCg8L0ysCmr214cQYphisvr4EbHw
# 👆

RUN pip install -r src/requirements.txt

CMD chainlit run src/app.py