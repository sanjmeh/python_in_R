import sys
import logging
import datetime
import configparser
import os.path
import time
import paho.mqtt.client as mqtt
from paho.mqtt.client import Client, MQTTv311, MQTTv5, MQTTv31
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

conf_file = sys.argv[1] if len(sys.argv) > 1 else 'config.ini'
config = configparser.ConfigParser()
if os.path.isfile(conf_file):
    config.read(conf_file)
else:
    raise FileNotFoundError(conf_file+" does not exist")

# mqtt settings
broker_address = str(config['MQTT_BROKER']['BrokerAddress'])
client_id = str(config['MQTT_BROKER']['ClientID'])
topics = str(config['MQTT_BROKER']['Topics'])
topics = topics.split(',') if ',' in topics else [topics]
qos_level = int(config['MQTT_BROKER']['QoS'])
mqtt_version = int(config['MQTT_BROKER']['MQTTVersion'])
clean_session_flag = False if str(
    str(config['MQTT_BROKER']['CleanSession'])).lower() == "false" else True
# logging settings
log_path = str(config['LOGGING']['Path'])
rotating_by_size = str(config['LOGGING']['RotateBySize']) == 'True'
rotating_log_backup_count = int(config['LOGGING']['FileCount'])
timed_rotating_unit = str(config['LOGGING']['RatationTimeUnit'])
timed_rotating_interval = int(config['LOGGING']['RatationTimeInterval'])
size_rotating_bytes = int(config['LOGGING']['RotationBytes'])
start_point = datetime.time(int(config['LOGGING']['RotationTimeStartHour']), int(config['LOGGING']['RotationTimeStartMinute']), int(config['LOGGING']['RotationTimeStartSecond'])) \
    if str(config['LOGGING']['RotationTimeStart']) == 'True' else None
current_time = int(time.time())

# setup stream logger
streamer = logging.getLogger('streamer')
streamHandler = logging.StreamHandler()
streamer.setLevel(logging.DEBUG)
streamer.addHandler(streamHandler)
# setup the handler for logging
handler = RotatingFileHandler(log_path, maxBytes=size_rotating_bytes, backupCount=rotating_log_backup_count) \
    if rotating_by_size is True else \
    TimedRotatingFileHandler(log_path, when=timed_rotating_unit,
                             interval=timed_rotating_interval, backupCount=rotating_log_backup_count, atTime=start_point)
if rotating_by_size is False:
    streamer.info('next log rotation: {}'.format(datetime.datetime.fromtimestamp(
        handler.computeRollover(int(time.time())))))

# setup the logging itself by adding the handler, path and minimum log level
filer = logging.getLogger('filer')
filer.setLevel(logging.INFO)
filer.addHandler(handler)

# this is a map of the different MQTT versions
mqtt_version_map = {
    "3": MQTTv31,
    "4": MQTTv311,
    "5": MQTTv5
}

# with this line the actual version implementation is picked from the map above based on the mqtt_version setting you choose
prot = mqtt_version_map[str(mqtt_version)]

# this is the funtion that is called on every successful subscription (it just logs atm)


def on_subscribe(client: Client, userdata, mid, granted_qos, *args):
    streamer.info(
        "Client: "+client._client_id.decode('utf8')+" - Subscribed id: "+str(mid) + " with qos " + str(granted_qos[0]))


# this is the funtion that is called on every successful message delivery (it just logs atm)
def on_message(client, userdata, msg, *args):
    filer.info("topic: "+str(msg.topic)+" - " + msg.payload.decode('utf8'))


def on_connect(client, userdata, flags, rc, *args):
    if rc == 0:
        client.connected_flag = True  # set flag
        for t in topics:
            client.subscribe(t.strip(), qos=qos_level)
    else:
        streamer.info(
            "Client: "+client._client_id.decode('utf8')+"("+rc+") connected at "+str(datetime.datetime.now()))
        client.reconnect()
    # now the subscriptions are actually started by using the specified topics


def on_disconnect(client, *args):
    streamer.info(
        "Client: "+client._client_id.decode('utf8')+" disconnected at "+str(datetime.datetime.now()))


# this is the client initialization using the specified client_id as client_id, clean_sessio and the picked protocol
client = mqtt.Client(client_id, clean_session=clean_session_flag,
                     protocol=prot)  # create new instance
streamer.info(client_id + " connecting to "+broker_address)

try:
    # the following to lines basically assign the functions from above to the actions that can happen like, subscribe and receive a message
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    # here the client actually connects to the broker
    client.connect(broker_address, keepalive=360)

    # this line is part of the MQTT paho library that needs to be made so that the client keeps up the connection with the broker
    client.loop_forever(timeout=10)
except KeyboardInterrupt:
    # catching kill and ctrl+c
    streamer.info("KeyboardInterrupt has happened")
except Exception as err:
    # catching errors
    e = sys.exc_info()[0]
    streamer.info("Error: "+str(e))
    streamer.info("Error Object: "+str(err)+" has happened")
finally:
    # here we do a disconnect in case of an error, so this happens whenever the program is closed or has a major error, so a non sophisticated error handling :)
    streamer.info(
        "Program ended. Disconnecting "+client._client_id.decode('utf8'))
    client.disconnect()