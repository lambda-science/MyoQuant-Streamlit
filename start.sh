#!/bin/bash

DIR=$(readlink -f $0 | xargs dirname)
PIDPATH=$DIR/.pid

if [ -f $PIDPATH ]; then
	kill $(cat $PIDPATH)
fi

nohup streamlit run Home.py --server.baseUrlPath /MyoQuant --browser.serverAddress lbgi.fr --server.fileWatcherType none --browser.gatherUsageStats false --logger.level warning > log 2>&1 &
echo $! > $DIR/pid
