# GQL is for consuming GraphQL APIs
# Graphene is for building GraphQL APIs

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

commit_data = client.execute(gql_query_commits)['viewer']['repositories']

df = pd.DataFrame(pd.json_normalize(commit_data['edges'], record_path=['node', 'refs', 'edges', 'node', 'target', 'history', 'edges'], meta=['node']))
df.head()

df['committedDateCET'] = pd.to_datetime(df['node.committedDate']).dt.tz_convert('Europe/Rome')
df['committedWeekDay'] = pd.to_datetime(df['committedDateCET']).dt.weekday # Monday=0, Sunday=6
df['committedYear'] = pd.to_datetime(df['committedDateCET']).dt.year
df['committedHourCET'] = pd.to_datetime(df['committedDateCET']).dt.hour  # .dt.floor('h')


work_day = [0, 1, 2, 3, 4]
df[(df['committedYear'] == 2021) & (df['committedWeekDay'].isin(work_day))].groupby(df["committedHourCET"]).count()

df.to_csv('data.csv')
