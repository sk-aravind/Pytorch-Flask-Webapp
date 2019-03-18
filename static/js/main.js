$(document).ready(function () {
  
  $('#btn-predict').hide();
  $('#imgpreview').hide();

  // Upload Preview
  function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#imgpreview')
                .attr('src', e.target.result)
                .height(300);
        };
        reader.readAsDataURL(input.files[0]);
    }
  }
  $("#imageUpload").change(function () {
      $('#imgpreview').show();
      $('#btn-predict').show();
      readURL(this);
  });

  // Predict
  $('#btn-predict').click(function () {
      // Show loading animation
      $(this).hide();
   
  });


});
