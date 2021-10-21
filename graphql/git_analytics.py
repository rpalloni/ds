import requests

# get github data
repo_list_url = "https://api.github.com/users/rpalloni/repos"
repo_comm_url = "https://api.github.com/repos/rpalloni/{name}/commits"

repo_list = requests.get(repo_list_url).json() # too many data
repo_name = []
for repo in repo_list:
    repo_name.append(repo['name'])

for name in repo_name:
    comm_list = requests.get(f'https://api.github.com/repos/rpalloni/{name}/commits').json() # too many data
    
'''    
GraphQL is a query language for an API.
It provides a standard way to:
   * describe data provided by a server in a statically typed Schema
   * request data in a Query which exactly describes your data requirements and
   * receive data in a Response containing only the data you requested.
'''

import graphene
