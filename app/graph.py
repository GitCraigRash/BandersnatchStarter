from altair import Chart, Tooltip
import altair as alt
def chart(df, x, y, target) -> Chart:
    graph= Chart(
        df,
        title=(f"{y} by {x} for {target}"),
        autosize = "pad",
        background = "#252525",
        padding = {"left":50,"top":35,"right":50,"bottom":35}
    ).mark_circle(size=100).encode(
        x= x,
        y= y,
        color=target,
        tooltip=Tooltip(df.columns.to_list())
    ).properties(
        width=450,
        height=470,
        ).configure_legend(
        titleColor = "#aaaaaa",
        labelColor='#aaaaaa',
        
)
    graph = graph.properties(
        title=alt.TitleParams(text=f"{y} by {x} for {target}",dy=-30,color="#aaaaaa", fontSize=25)
        
        )
    graph = graph.configure_view(
        stroke = "#aaaaaa",
        strokeOpacity = 0.1
    )
    graph = graph.configure_axis(
        tickCount = 20,
        titleColor = "#aaaaaa",
        titlePadding = 20,
        tickMinStep = 20,
        gridColor = '#aaaaaa',
        tickColor = "#aaaaaa",
        tickOpacity = 0.1,
        labelColor = "#aaaaaa",
        
        gridOpacity = 0.1
    )
    return graph

if __name__ == '__main__':
    pass
