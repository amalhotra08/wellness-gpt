
  function scrollToBottom() {
    const chatContainer = document.querySelector('.chat-container');
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
  document.addEventListener("DOMContentLoaded", function () {
    scrollToBottom();
    const overlay = document.getElementById("overlay");


    const username_element = document.getElementById("usernameInput")

    // Check if the username is already stored in localStorage
    const savedUsername = localStorage.getItem("username");

    if (savedUsername) {
      // If the username is found, hide the popup
      overlay.style.display = "none";
      username_element.textContent = savedUsername;
      fetch("/username-endpoint", {
        method: "POST",
        body: JSON.stringify({ username: savedUsername }),
        headers: {
          "Content-Type": "application/json"
        }
      }).then(response => response.json()).then(data => {
        console.log(data);
      }).catch(error => {
        console.error("Error:", error);
      })
    } else {
      // If the username is not found, display the popup
      overlay.style.display = "flex";
    }
    setChatbotAvatar();
  });

  // Endpoint for username form
  document.getElementById("username-form").addEventListener("submit", function(event) {
    event.preventDefault();
    const username = usernameInput.value;

    if (username.trim() !== "") {
      // Store the username in localStorage
      localStorage.setItem("username", username);
      // Hide the popup
      overlay.style.display = "none";
      username_element.textContent = savedUsername;
      fetch("/username-endpoint", {
        method: "POST",
        body: JSON.stringify({ username: savedUsername }),
        headers: {
          "Content-Type": "application/json"
        }
      }).then(response => response.json()).then(data => {
        console.log(data);
      }).catch(error => {
        console.error("Error:", error);
      })
    } else {
      alert("Please enter a username.");
    }
  })



  // Get the image upload input and the avatar image element

  const imageUploadInput = document.getElementById("image-upload-input");
  const avatarImage = document.getElementById("avatar-image");

  // Update the image upload handling code to save the avatar URL
  imageUploadInput.addEventListener("change", (event) => {
      const file = event.target.files[0];

      if (file) {
      const reader = new FileReader();

      reader.onload = (e) => {
          avatarImage.src = e.target.result;

          // Save the uploaded avatar URL to local storage
          localStorage.setItem("uploadedAvatar", e.target.result);

          // Update the chatbot's avatar as well
          setChatbotAvatar();
      };

      reader.readAsDataURL(file);
      }
  });


  // Function to set the chatbot's avatar image from local storage
  function setChatbotAvatar() {
    const chatbotAvatarImage = document.getElementById("chatbot-avatar-image");
    const uploadedAvatarURL = localStorage.getItem("uploadedAvatar");

    if (uploadedAvatarURL) {
      chatbotAvatarImage.src = uploadedAvatarURL;
    }
  }

  setChatbotAvatar();
  document.getElementById('uploadButton').addEventListener('click', function() {
    const fileInput = document.getElementById('genomic-info-upload-input');

    if (fileInput.files.length > 0) {
      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append('genomicFile', file);

      fetch('/genomic_data', {
        method: 'POST',
        body: formData
      })
      .then(response => {
        if (response.ok) {
          // Show the confirmation message
          document.getElementById('confirmation-message').style.display = 'block';

          // Hide the confirmation message after 3 seconds
          setTimeout(function() {
            document.getElementById('confirmation-message').style.display = 'none';
          }, 3000);
          return response.json();
        }
        throw new Error('Network response was not ok.');
      })
      .then(data => {
        // Handle the response data from the backend if needed
        console.log('File uploaded successfully:', data);
      })
      .catch(error => {
        // Handle errors
        console.error('There was a problem with the upload:', error);
      });
    } else {
      console.log('Please select a file to upload.');
    }
  }); 