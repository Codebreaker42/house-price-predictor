import logging
import os 

# log directory creation 
log_dir="logs"
os.makedirs(log_dir, exist_ok= True )

# logging configuratation 
logger= logging.getLogger("data_ingestion")
logger.setLevel('DEBUG')

# making handler 
console_handler= logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path= os.path.join(log_dir, 'data_ingestion')
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# setting the format of logging message 
formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug("logging starts here")