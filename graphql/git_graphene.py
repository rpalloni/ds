import requests
from graphene import Schema, Field, String, Int, ObjectType

class Folder(ObjectType):
    name = String()
    ncommits = Int()


class Query(ObjectType):

    folder = Field(Folder)

    def resolve_folder(root, info, query, headers):
        data = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)
        return Folder(data)


schema = Schema(query=Query)
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

result = schema.execute(gql_query)
print(result.data["folder"])
