#!/bin/bash
echo "Fixing user_id and permissions"
set -x

mkdir -p /run/bisque/data/{server_cache,uploads}

user=$(getent passwd bisque)
if [ ! -z "$user" -a "$user" != "$BISQUE_USER" ] ; then
    usermod bisque -l $BISQUE_USER
fi

#usermod -d /source $BISQUE_USER
if [ $(id -u $BISQUE_USER) != "$BISQUE_UID" ] ; then
    # usermod chnage file permission in home dir (which is set above)
    usermod -u $BISQUE_UID $BISQUE_USER
fi

mkdir -p /run/bisque/data /run/bisque/local
chown $BISQUE_USER /run/bisque  /run/bisque/data /run/bisque/local /run/bisque/data/{server_cache,uploads}
chown $BISQUE_USER -R /source/modules

# Fix sqlite database permissions if it exists
if [ -f /source/bisque.db ]; then
    chown $BISQUE_USER:$BISQUE_USER /source/bisque.db
    chmod 664 /source/bisque.db
fi

rsync -a /source/public/ /usr/lib/bisque/lib/python3.10/site-packages/bq/core/public/
chown $BISQUE_USER -R /usr/lib/bisque/lib/python3.10/site-packages/bq/core/public
# change all except data dir
#find /source -user root ! -name /source/data  | xargs  chown $BISQUE_USER
