FROM docker.elastic.co/elasticsearch/elasticsearch:8.13.0

# Copy initialization script
COPY init.sh /usr/local/bin/es-init.sh
USER root
RUN chmod +x /usr/local/bin/es-init.sh
USER elasticsearch

EXPOSE 9200 9300
