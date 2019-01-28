window.onscroll = function (e) {
    var vertical_position = 0;
    if (pageYOffset)//usual
      vertical_position = pageYOffset;
    else if (document.documentElement.clientHeight)//ie
      vertical_position = document.documentElement.scrollTop;
    else if (document.body)//ie quirks
      vertical_position = document.body.scrollTop;
  
    var your_div = document.getElementById('whiteout');
    your_div.top = (vertical_position + 200) + 'px';//200 is arbitrary.. just to show you could now position it how you want
  }

//   window.onload = function(){

//     var sideButtons = document.getElementsByClassName("leftNavSection");
//     var numElements = sideButtons.length;
//     console.log(numElements);
//     if(numElements > 0){
 
//      var lastActiveButton = sideButtons[0]; //SET TO INTRODUCTION
//      console.log(lastActiveButton.length);
//       for(var i = 0; i < numElements; i++){
//         sideButtons[i].addEventListener("click", function(){
//           console.log(this);
//           this.classList.add("active");
//           lastActiveButton.classList.remove("active");
          
//            for(var j = 0; j < numElements; j++){ //In this context numElements is the number of pages
//              document.getElementById("page" + j + 1).setAttribute("display", "none");
//              //Make sure nothing else is showing.
//            }
 
//            this.classlist.add("active");
//            if(this.classlist.contains("page1")){
//              document.getElementById("page1").setAttribute("display", "block");
//            }
//            if(this.classlist.contains("page2")){
//              document.getElementById("page2").setAttribute("display", "block");
//            }
//            if(this.classlist.contains("page3")){
//              document.getElementById("page3").setAttribute("display", "block");
//            }
//            if(this.classlist.contains("page4")){
//              document.getElementById("page4").setAttribute("display", "block");
//            }
//            if(this.classlist.contains("page5")){
//              document.getElementById("page5").setAttribute("display", "block");
//            }
//            if(this.classlist.contains("page6")){
//              document.getElementById("page6").setAttribute("display", "block");
//            }
//            if(this.classList.contains("page7")){
//              document.getElementById("page7").setAttribute("display", "block");
//            }
 
           
//         });
//       }
//     }
//  }
