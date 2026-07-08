import plotly.express as px
import pandas as pd


def create_model_comparison_chart(
    comparison_df,
    metric
):

    fig = px.bar(

        comparison_df,

        x="Model",

        y=metric,

        color=metric,

        text=metric,

        title=f"Model Comparison ({metric})"

    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig.update_layout(

        xaxis_title="Model",

        yaxis_title=metric,

        template="plotly_white"

    )

    return fig


def create_cv_chart(
    cv_df
):

    fig = px.bar(

        cv_df,

        x="Model",

        y="CV Mean",

        color="CV Mean",

        text="CV Mean",

        title="Cross Validation Comparison"

    )

    fig.update_traces(

        texttemplate="%{text:.4f}",

        textposition="outside"

    )

    fig.update_layout(

        template="plotly_white"

    )

    return fig


def create_feature_importance_chart(
    feature_df
):

    fig = px.bar(

        feature_df,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        title="Feature Importance"

    )

    fig.update_layout(

        template="plotly_white"

    )

    return fig


def create_residual_chart(
    plot_df
):

    fig = px.scatter(

        plot_df,

        x="Predicted",

        y="Residual",

        title="Residual Plot"

    )

    fig.add_hline(
        y=0
    )

    return fig