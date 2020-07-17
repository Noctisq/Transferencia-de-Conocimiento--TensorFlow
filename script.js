let model;

const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();


var count = 0;
var net ;
var webcam;
async function app(){
	console.log("Cargando modelo de identificacion de imagenes");
  net= await mobilenet.load();
	console.log("Carga terminada")
  //clasificamos la imagen de carga


  

  
  //obtenemos datos del webcam
	webcam = await tf.data.webcam(webcamElement);
  //y los vamos procesando
  while (true) {
    const img = await webcam.capture();
    
    const result = await net.classify(img);
    
    const activation = net.infer(img, 'conv_preds');
    var result2; 
    try {
      result2 = await classifier.predictClass(activation);  
    } catch (error) {
      result2 = {};
    }
    
    const classes = ["Sin Entrenar", "Celular", "Audifonos" , "Cartera", "Videojuego","Pluma"]

    //Puede ustar esta linea para ver como el modelo de google predice la imagen en su webcam
    // document.getElementById('console').innerText = `
    //   prediccion: ${result[0].className}\n
    //   probabilidad: ${result[0].probability}
    // `;

    try {
      document.getElementById("console2").innerText = `
    prediccion: ${classes[result2.label]}\n
    probabilidad: ${result2.confidences[result2.label]}
    `;
    } catch (error) {
      document.getElementById("console2").innerText="Sin entrenar";
    }
    


    
    img.dispose();

    await tf.nextFrame();
  }
}




async function addExample (classId) {
  const img = await webcam.capture();
  const activation = net.infer(img, true);
  classifier.addExample(activation, classId);
  //liberamos el tensor
  img.dispose()
}

app()