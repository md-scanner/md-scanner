import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from qdrant_client import QdrantClient
from common import *

# Reference:
# https://stackabuse.com/guide-to-multidimensional-scaling-in-python-with-scikit-learn/

print(f"[DbVisualize] Initializing the DB...")
db_client = QdrantClient(path=FSC_DB_PATH)

print(f"[DbVisualize] DB initialized")

points_embedding = []
points_font = []

page = None
largest_id = 0
while True:  # Iterate over every page of the DB (all points)
    # QDrant scroll is described here:
    # # https://stackabuse.com/guide-to-multidimensional-scaling-in-python-with-scikit-learn/
    page = db_client.scroll(
        collection_name=FSC_DB_COLLECTION_NAME,
        with_payload=True,
        with_vectors=True,
        offset=largest_id + 1,
        limit=1000
    )

    elements, _ = page

    if len(elements) == 0:
        break

    print("Queried %d points from DB, largest ID: %d..." % (len(elements), largest_id))

    largest_id = max(map(lambda x: x.id, elements))

    points_embedding.extend(x.vector for x in elements)
    points_font.extend(x.payload['font'] for x in elements)

# Project embeddings into the 2D space (preserving the Euclidean distances)
mds = MDS(random_state=0, normalized_stress='auto')
_2d_points = mds.fit_transform(points_embedding)

x_coords, y_coords = zip(*_2d_points)
colors = ['#{:06x}'.format(hash(font) & 0xFFFFFF) for font in points_font]  # Get a unique color per font

# Show the 2d points on a chart
plt.scatter(x_coords, y_coords, color=colors, label='Points', s=30)

plt.title("FSC DB visualization")
plt.xlabel('x')
plt.ylabel('y')

plt.show()
