#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:02:32 2020

@author: eduardo
"""

from flask import Flask, render_template, request, redirect, session, flash, url_for
import os
import pandas as pd
from datetime import datetime


# Iniciando o Flask
os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
app = Flask(__name__)


@app.route("/index")
@app.route("/")
def index():
    return render_template(
        "template.html", titulo="Elaine Alves", botao_lateral="teste"
    )


app.run(debug=True)

# spa.finalizar()
