from FileService import add_xmls_in_renate
from AnnotationService import create_and_add_annotations
from AuthService import get_authenticated_session


s = get_authenticated_session()
add_xmls_in_renate(s, "10d7aac1-a01e-43f7-be42-8280f16160c5")

#only for physics papers
#considering only physics collection
create_and_add_annotations(s, "10d7aac1-a01e-43f7-be42-8280f16160c5")

#get_xml()

#pp = Flask(__name__)

#call function to create xml annotations
#call function to annotate xml files


#@app.route('/create_xml', methods=['GET'])
#def create_xml():
    #fileName = request.args.get('fileName')

    #if not fileName:
        #return jsonify({"error": "Invalid or missing fileName"}), 400
    #print(fileName)
    #try:
        #xml_str = process_file(fileName)
        #return pdf_to_xml(fileName)
        #return xml_str, 200, {'Content-pcx8f. cuyuupType': 'application/xml'}
    #except FileNotFoundError:
        #return jsonify({"error": "File not found"}), 404
    #except Exception as e:
        #return jsonify({"error": str(e)}), 500

#@app.route('/')
#def welcome():
    #return "This API converts PDF articles to XML"

#if __name__ == '__main__':
    #app.run(debug=True)