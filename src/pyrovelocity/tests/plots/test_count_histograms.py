from pyrovelocity.plots import plot_spliced_unspliced_histogram


def test_plot_spliced_unspliced_histogram(adata_preprocessed, tmp_path):
    chart = plot_spliced_unspliced_histogram(adata_preprocessed)
    chart.save(tmp_path / "spliced_unspliced_histogram.pdf")
    chart.save(tmp_path / "spliced_unspliced_histogram.svg")
    chart.save(tmp_path / "spliced_unspliced_histogram.png")
    chart.save(tmp_path / "spliced_unspliced_histogram.html")
