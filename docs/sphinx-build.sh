#!/bin/bash

echo "***********************************************************************"
echo "                Starting the sphinx script"
echo "***********************************************************************"
echo "            Creating plugin API files and html pages .."

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# members will document all modules
# undoc keeps modules without docstrings
# show-inheritance displays a list of base classes below the class signature
export SPHINX_APIDOC_OPTIONS='members,show-inheritance,undoc-members'

# Remove directory for api so that there are no obsolete files
rm -rf $DIR/source/api/
rm -rf $DIR/build/

# sphinx-apidoc [options] -o <outputdir> <sourcedir> [pathnames to exclude]
# -f, --force. Usually, apidoc does not overwrite files, unless this option is given.
# -e, --separate. Put each module file in its own page.
# -T, --no-toc. Do not create a table of contents file.
# -E, --no-headings. Do not create headings for the modules/packages. This is useful, for example, when docstrings already contain headings.
# -t template directory
# add -Q to suppress warnings

# sphinx-apidoc generates source files that use sphinx.ext.autodoc to document all found modules
sphinx-apidoc -feT -t=$DIR/source/_templates -o $DIR/source/api $DIR/../httomo_backends

# build yaml templates here:
python $DIR/../httomo_backends/scripts/yaml_templates_generator.py -i $DIR/../httomo_backends/methods_database/packages/backends/tomopy/tomopy_modules.yaml -o $DIR/build/yaml_templates/tomopy
python $DIR/../httomo_backends/scripts/yaml_unsupported_tomopy_remove.py -t $DIR/build/yaml_templates/tomopy -l $DIR/../httomo_backends/methods_database/packages/backends/tomopy/tomopy.yaml
python $DIR/../httomo_backends/scripts/yaml_templates_generator.py -i $DIR/../httomo_backends/methods_database/packages/backends/httomolibgpu/httomolibgpu_modules.yaml -o $DIR/build/yaml_templates/httomolibgpu
python $DIR/../httomo_backends/scripts/yaml_templates_generator.py -i $DIR/../httomo_backends/methods_database/packages/backends/httomolib/httomolib_modules.yaml -o $DIR/build/yaml_templates/httomolib

# Append yaml link to rst files
python -m source.yaml_doc_generator

# sphinx-build [options] <sourcedir> <outputdir> [filenames]
# -a Write all output files. The default is to only write output files for new and changed source files. (This may not apply to all builders.)
# -E Don’t use a saved environment (the structure caching all cross-references), but rebuild it completely. The default is to only read and parse source files that are new or have changed since the last run.
# -b buildername, build pages of a certain file type
sphinx-build -a -E -b html $DIR/source/ $DIR/build/


echo "***********************************************************************"
echo "                          End of script"
echo "***********************************************************************"

