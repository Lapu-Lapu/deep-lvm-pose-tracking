#!/bin/bash
pandoc -s --bibliography=literature.bib -o notes.pdf notes.org
