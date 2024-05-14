FROM elix-kmol:1.1.9.1

RUN echo "conda activate kmol" >> ~/.bashrc

SHELL [ "/bin/bash", "--login", "-c" ]

RUN conda install -c conda-forge -c ljn917 molvs rdchiral_cpp
RUN pip install epam.indigo quantulum3

RUN mkdir -p /opt/elix/kmol/ReTReKpy

WORKDIR /opt/elix/kmol/ReTReKpy

RUN cd /opt/elix/kmol/ReTReKpy

ENTRYPOINT ["/opt/envs/kmol/bin/python"]
