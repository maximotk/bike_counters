import folium
from streamlit_folium import st_folium

def make_counter_map(data):
    map_data = (
        data[["counter_name", "latitude", "longitude"]]
        .drop_duplicates("counter_name")
        .dropna(subset=["latitude", "longitude"])
    )
    if map_data.empty:
        return

    m = folium.Map(location=map_data[["latitude", "longitude"]].mean().values.tolist(), zoom_start=12)
    for _, row in map_data.iterrows():
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=row["counter_name"],
            icon=folium.Icon(color="blue", icon="bicycle", prefix="fa")
        ).add_to(m)
    st_folium(m, width=1000, height=450)
