"""Accuracy-vs-throughput harness (accuracy half is CPU-runnable; see docs/DESIGN.md §9).

Throughput benchmarking belongs to scripted rented-GPU sessions and never runs
here. The *accuracy* characterization (relative error vs size and conditioning
against the mpmath reference, docs/DESIGN.md §6) is pure CPU work and lives in this
package.
"""
