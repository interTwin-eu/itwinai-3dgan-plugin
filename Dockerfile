# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /usr/src/app

# Install itwinai
COPY pyproject.toml pyproject.toml
COPY src src
RUN pip install --upgrade pip \
    && pip install --no-cache-dir .
