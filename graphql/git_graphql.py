###########################################################################
########################## GitHub REST API ################################
###########################################################################

# "https://api.github.com/users/rpalloni/repos"
# "https://api.github.com/repos/rpalloni/{name}/commits"

import re
import requests

token = 'git_token'
headers = {'Authorization': 'token ' + token} # set token

repo_data = requests.get('https://api.github.com/users/rpalloni/repos', headers=headers).json() # too many data, just need name

repo_ncommit = {}
for repo in repo_data:
    repo_name = repo['name']
    repo_url = f'https://api.github.com/repos/rpalloni/{repo_name}/commits?per_page=1'
    repo_ncommit[repo_name] = re.search('\d+$', requests.get(repo_url).links['last']['url']).group() # too many data, just need n commits

print(repo_ncommit)


###########################################################################
####################### GitHub GraphQL API ################################
###########################################################################

# https://api.github.com/graphql

'''
GraphQL is a query language for an API.
It provides a standard way to:
   * describe data provided by a server in a statically typed Schema
   * request data in a query which exactly describes the data requirements
   * receive data in a response containing only the data requested
'''

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

repo_data = requests.post('https://api.github.com/graphql', 
            headers=headers, json={'query': gql_query}
            ).json()['data']['viewer']['repositories']['edges'] # only requested data

repo_ncommit = {}
for repo in repo_data:
  repo_ncommit[repo['node']['name']] = repo['node']['refs']['edges'][0]['node']['target']['history']['totalCount']
print(repo_ncommit)