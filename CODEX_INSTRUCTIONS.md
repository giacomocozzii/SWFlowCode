# Instructions for Codex (VS Code)

Open this extracted project folder in VS Code.

## First task
1. Read README.md and inspect app.py, density.py, flow.py, and tests/.
2. Create a local `.venv`, install `requirements.txt`, and run `python -m pytest -q`.
3. Start the app with `python -m streamlit run app.py`.
4. Confirm that both initial and target density formulas can be edited independently.
5. Test the default 1000-particle / 384-direction simulation, its animation, and the energy plot.
6. Do not change the mathematical definition or publish anything to GitHub without explicit approval.

## Prompt to paste into Codex

Please inspect this Sliced Wasserstein Flow Lab project and run it locally.
Use a Python virtual environment. Install the dependencies from requirements.txt,
run all tests, and launch Streamlit. Check the two independent LaTeX density
inputs and test the default Gaussian-to-translated-Gaussian experiment with
1000 particles and 384 projection directions. Report any runtime or UI issues.
Keep the empirical Wasserstein gradient convention in flow.py unchanged
unless you explain and verify the mathematical correction.
Do not push or publish to GitHub without asking me first.

## Notes
- The formulas are parsed by a restricted arithmetic parser, not a general LaTeX interpreter.
- Densities are normalized on a chosen finite computational rectangle.
- This is an empirical finite-direction numerical experiment, not an exact PDE solver.
- The code is intended for a separate public demo repository, not the private SpEl project.
