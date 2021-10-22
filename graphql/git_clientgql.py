# GQL is for consuming GraphQL APIs
# Graphene is for building GraphQL APIs

from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
#from gql.transport.aiohttp import AIOHTTPTransport # async request

headers = {'Authorization': 'token ' + token} # set token

client = Client(transport=RequestsHTTPTransport(url='https://api.github.com/graphql', headers=headers))

gql_query = gql(
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

result = client.execute(gql_query)
print(result)
