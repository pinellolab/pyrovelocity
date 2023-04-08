import altair as alt


def custom_theme():
    font = "LMSans10"

    return {
        "config": {
            "title": {"fontSize": 16, "font": font},
            "axis": {
                "titleFontSize": 18,
                "labelFontSize": 18,
                "labelFont": font,
                "titleFont": font,
            },
            "header": {"labelFontSize": 14, "labelFont": font},
            "legend": {"titleFontSize": 14, "labelFontSize": 12, "font": font},
            "mark": {
                "fontSize": 14,
                "tooltip": {"content": "encoding"},
                "font": font,
            },
        }
    }
