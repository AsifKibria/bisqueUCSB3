#!/bin/bash
set -x

echo "Adding Bisque User "

adduser --uid $BISQUE_UID --disabled-login --gecos "Bisque" --home /home/bisque --shell /bin/bash $BISQUE_USER
echo "$BISQUE_USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# *required* all COPY and ADD are by root https://github.com/moby/moby/issues/6119
chown -R $BISQUE_USER /builder /source
