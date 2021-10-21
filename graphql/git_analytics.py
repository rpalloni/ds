import re
import requests

# get github repo data
repo_list_url = "https://api.github.com/users/rpalloni/repos"
repo_comm_url = "https://api.github.com/repos/rpalloni/{name}/commits"


headers = {'Authorization': 'token ' + token} # set token
repo_data = requests.get(repo_list_url, headers=headers).json() # too many data, just need name

repo_ncommit = {}
for repo in repo_data:
    repo_name = repo['name']
    repo_url = f'https://api.github.com/repos/rpalloni/{repo_name}/commits?per_page=1'
    repo_ncommit[repo_name] = re.search('\d+$', requests.get(repo_url).links['last']['url']).group() # too many data, just need n commits

repo_ncommit


'''    
GraphQL is a query language for an API.
It provides a standard way to:
   * describe data provided by a server in a statically typed Schema
   * request data in a query which exactly describes the data requirements
   * receive data in a response containing only the data requested
'''

# github graphql api

gql_query = """
{
  viewer {
    repositories(first: 30) {
      edges {
        node {
          name
          refs(first: 100, refPrefix: "refs/heads/") {
            edges {
              node {
                name
                target {
                  ... on Commit {
                    history(first: 0) {
                      totalCount
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

data = requests.post('https://api.github.com/graphql', headers=headers, json={'query': query}).json() # only requested data
data['data']['viewer']['repositories']['edges']

