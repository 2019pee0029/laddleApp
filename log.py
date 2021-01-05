import sys
import logging
import logging.handlers
import os

# the main log
log = logging.getLogger('Laddle-Id-detection')

# the events log
events_log = logging.getLogger('sensu-trapd-events')

def configure_log(log, log_file, log_level, foreground=None):
    # Clear existing log handlers
    log.handlers = []

    # Configure Log Formatting
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    filehandler = logging.handlers.WatchedFileHandler(log_file)
    filehandler.setFormatter(formatter)
    log.addHandler(filehandler)

    # Configure Logging Level
    try:
        log.setLevel(getattr(logging, log_level))
    except AttributeError:
        log.warn("Unknown logging level: %s" % (log_level))
        log.setLevel(logging.INFO)

    # Configure Foreground Logging
    if foreground:
        streamhandler = logging.StreamHandler(sys.stdout)
        streamhandler.setFormatter(formatter)
        log.addHandler(streamhandler)

def setupTimeRotatedLog(filename,logger):
    filepath = os.path.join(os.getcwd(),"logs",filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    log_handler =logging.handlers.TimedRotatingFileHandler(filepath, when="midnight", interval=1)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    log_handler.suffix = "%Y%m%d"
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)


def configure_events_log(log, log_file):
    # Clear existing log handlers
    log.handlers = []

    # Configure Log Formatting
    formatter = logging.Formatter('%(asctime)s|%(message)s')
    filehandler = logging.handlers.WatchedFileHandler(log_file)
    filehandler.setFormatter(formatter)
    log.addHandler(filehandler)

    # Configure Logging Level
    log.setLevel(logging.INFO)
