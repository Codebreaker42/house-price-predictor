import os
import logging 

# ensure the log directory exist
log_dir='logs'
os.makedirs(log_dir, exist_ok= True )

# setting up logger 
logger= logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# making handler 
console_handler= logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file handler 
log_file_path= os.path.join(log_dir, 'data_preprocessing.log')
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# defining formatter 
formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s ')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adding handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug("data Preprocessing Logging starts here ")


def main():
    pass 

if __name__ == "__main__":
    main()