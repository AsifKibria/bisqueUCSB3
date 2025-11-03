#!/bin/bash

## NOTE: trailing slashes for rsync !
LOCAL=/run/bisque/local/workdir/
CACHE=/run/bisque/data/work_cache/

echo "Restoring workdir if possible"

if [ ! -d $LOCAL ] ; then
    echo "exiting: no local workdir $LOCAL"
    exit 0
fi

if [ -d $CACHE ] ; then
    echo "syncing: $CACHE -> $LOCAL"
    rsync -a  $CACHE $LOCAL
fi
