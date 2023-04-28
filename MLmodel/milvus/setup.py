# Run `python3` in your terminal to operate in the Python interactive mode.
from pymilvus import connections


connections.connect(
  alias="default", 
  user='username',
  password='password',
  host='localhost', 
  port='19530'
)