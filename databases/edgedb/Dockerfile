FROM edgedb/edgedb:4

# Copy schema files
COPY schema /dbschema
COPY migrate.sh /docker-entrypoint-initdb.d/migrate.sh

USER root
RUN chmod +x /docker-entrypoint-initdb.d/migrate.sh
