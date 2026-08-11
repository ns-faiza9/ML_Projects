from flask import Flask, render_template
from load_data import get_data_summary, load_data
from eda import run_eda

app = Flask(__name__, template_folder=".")


@app.route("/")
def index():
    # Landing page, no section selected yet
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():
    """Loads the dataset (server-side) and renders the summary into the page."""
    error = None
    summary = None
    try:
        summary = get_data_summary()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error,
    )


@app.route("/data")
def data_table():
    """Returns the full dataset rendered as an HTML table fragment."""
    try:
        df = load_data()
    except FileNotFoundError as e:
        return str(e), 500
    except Exception as e:
        return f"Unexpected error: {e}", 500

    columns = list(df.columns)
    rows = df.astype(str).values.tolist()
    return render_template("data_table.html", columns=columns, rows=rows)


@app.route("/eda")
def eda_page():
   error = None
   results = None
   try:
       results = run_eda()
   except FileNotFoundError as e:
       error = str(e)
   except Exception as e:
       error = f"Unexpected error: {e}"

   return render_template(
       "eda.html",
       active="eda",
       results=results,
       error=error,
   )


if __name__ == "__main__":
    app.run(debug=True)