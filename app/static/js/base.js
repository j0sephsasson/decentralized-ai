// subscribe to newsletter
async function subscribe(event) {
    var email = document.getElementById('new-sub').value;

    var fd = new FormData();
    fd.append('email', email);

    $.ajax({
        type: "POST",
        url: '/subscribe',
        data: fd,
        processData: false,
        contentType: false
    }).done(function (err, req, resp) {
        event.preventDefault();
        document.getElementById('response').innerHTML = resp.responseJSON.resp;
        document.getElementById('response').style.display = 'initial';
        setTimeout(function () {
            $('#response').fadeOut('slow');
        }, 7000);
    });
};

// CONTACT FORM
async function sendMail() {
    var name = document.getElementById('contact-form-name').value;
    var email = document.getElementById('contact-form-email').value;
    var subject = document.getElementById('contact-form-subject').value;
    var message = document.getElementById('contact-form-message').value;

    if (name == null || name == "", email == null || email == "", subject == null || subject == "", message == null || message == "") {
        alert("Please Fill All Required Fields");
        window.location.href = '/#contact'
    } else {
        document.getElementById('sendmailLoadBTN').style.visibility = 'visible';

        var fd = new FormData();
        fd.append('name', name);
        fd.append('email', email);
        fd.append('subject', subject);
        fd.append('message', message);

        $.ajax({
            type: "POST",
            url: '/send_message',
            data: fd,
            processData: false,
            contentType: false
        }).done(function (err, req, resp) {
            document.getElementById('sendmailLoadBTN').style.visibility = 'hidden';
            document.getElementById('contact-form-response').innerHTML = resp.responseJSON.resp;
            document.getElementById('contact-form-respdiv').style.display = 'initial';
            setTimeout(function () {
                $('#contact-form-respdiv').fadeOut('slow');
            }, 7000);
            document.getElementById('contact-form-name').value = null;
            document.getElementById('contact-form-email').value = null;
            document.getElementById('contact-form-subject').value = null;
            document.getElementById('contact-form-message').value = null;
        });
    };
};

// train progress bar
function train() {
    document.getElementById('trainBTN').style.display = 'none';
    document.getElementById('progress').style.display = 'initial';
};

// white paper alert for MVP
function whitePaper() {
    alert("White Paper Coming Soon!");
};

// register upload from user
function registerUpload() {
    document.getElementById("upload-label").style.backgroundColor = 'green';
}