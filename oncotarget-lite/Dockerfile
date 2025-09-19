FROM continuumio/miniconda3:24.1.2-0

# Create non-root user
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --shell /bin/bash --create-home ${USERNAME}

WORKDIR /workspace
COPY environment.yml /workspace/environment.yml
RUN conda env create -f environment.yml \
    && conda clean -afy

ENV PATH="/opt/conda/envs/oncotarget-lite/bin:${PATH}"

COPY . /workspace
RUN chown -R ${USERNAME}:${USERNAME} /workspace
USER ${USERNAME}

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
