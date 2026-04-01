FROM redis:7-alpine

# Copy custom config
COPY redis.conf /usr/local/etc/redis/redis.conf

# Copy init script
COPY init.sh /docker-entrypoint-initdb.d/init.sh
RUN chmod +x /docker-entrypoint-initdb.d/init.sh

EXPOSE 6379

CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]
