#!/bin/bash
set -x

confd -onetime -backend env

#mkdir -p /etc/bisque /run/bisque /run/bisque/staging
#chown $BISQUE_USER -R /etc/bisque /usr/share/bisque
#chown $BISQUE_USER /run/bisque /run/bisque/bqfeature  /run/bisque/external  /run/bisque/public /run/bisque/staging
#chown $BISQUE_USER /run/bisque/data /run/bisque/data/*

# NGINX Configure
mkdir -p /tmp/nginx
chmod ugo+rwx /tmp/nginx
#mkdir -p /var/nginx/www
#mkdir -p /var/nginx/store
#rsync -aL /run/bisque/public/ /var/nginx/www/
#chmod ugo+rwx /var/nginx /var/nginx/store
#chmod -R ugo+r /var/nginx/www

# Docker access for engine
if [ -e /var/run/docker.sock ] ; then
    fgrep docker /etc/group || addgroup docker
    groupmod -g $(ls -gn /var/run/docker.sock|awk '{print $3;}') docker
    usermod -aG docker $BISQUE_USER
fi

exec /usr/bin/supervisord  -c  /etc/supervisor/supervisord.conf
