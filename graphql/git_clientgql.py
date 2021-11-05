# GQL is for consuming GraphQL APIs
# Graphene is for building GraphQL APIs

import pytz # timezone
import pandas as pd
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
#from gql.transport.aiohttp import AIOHTTPTransport # async request

token = 'git_token'
headers = {'Authorization': 'token ' + token} # set token

client = Client(transport=RequestsHTTPTransport(url='https://api.github.com/graphql', headers=headers))

gql_query_repos = gql(
"""
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
)

repo_data = client.execute(gql_query_repos)['viewer']['repositories']['edges']

repo_ncommit = {}
for repo in repo_data:
  repo_ncommit[repo['node']['name']] = repo['node']['refs']['edges'][0]['node']['target']['history']['totalCount']
print(repo_ncommit)


gql_query_commits = gql(
    """
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
                    history {
                      edges {
                        node {
                          id
                          committedDate
                        }
                      }
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
)

commit_data = client.execute(gql_query_commits)['viewer']['repositories']['edges']

df = pd.DataFrame(pd.json_normalize(commit_data, ['node', 'refs', 'edges', 'node', 'target', 'history', 'edges'], meta=['node']))

df['committedDateCET'] = pd.to_datetime(df['node.committedDate']).dt.tz_convert('Europe/Rome')
df['committedYear'] = pd.to_datetime(df['committedDateCET']).dt.year
df['committedDateCET'] = pd.to_datetime(df['committedDateCET']).dt.hour


df.groupby(df["committedDateCET"]).count()

df[df['committedYear'] == 2021].groupby(df["committedDateCET"]).count()

df.to_csv('data.csv')
