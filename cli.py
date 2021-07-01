from jina.clients import Client

from jina import Document

def print_matches(req):  # the callback function invoked when task is done
    for idx, d in enumerate(req.docs[0].matches[:3]):  # print top-3 matches
        print(f'[{idx}]{d.score.value:2f}: "{d.text}"')

c = Client(port_expose=8080)
while True:
    c.search(Document(text=input()), on_done=print_matches)
