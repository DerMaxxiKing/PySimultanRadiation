# buid pyemblite

#FROM maxxiking/embree3:1.0.2 AS compile-image
#
#FROM ubuntu:20.04
#
#COPY --from=compile-image /usr/local/lib/embree-3.13.1.x86_64.linux /usr/local/lib/embree-3.13.1.x86_64.linux
#COPY resources/pyemblite-0.0.1-cp38-cp38-linux_x86_64.whl /tmp/pyemblite-0.0.1-cp38-cp38-linux_x86_64.whl
#
#RUN apt-get update \
#    && apt-get install -y --no-install-recommends python3.8 \
#      python3-pip \
#    && apt-get autoremove -y  \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/*
#
#ENV LD_LIBRARY_PATH=/usr/local/lib/embree-3.13.1.x86_64.linux/lib:
#
#RUN echo "source /usr/local/lib/embree-3.13.1.x86_64.linux/embree-vars.sh" >> /etc/bash.bashrc \
#    && /bin/bash -c 'source ~/.bashrc' \
#    && cd /tmp/ \
#    && pip3 install pyemblite-0.0.1-cp38-cp38-linux_x86_64.whl


FROM maxxiking/pyemblite

COPY reqiurements.txt /tmp/reqiurements.txt

RUN pip3 install reqiurements.txt
