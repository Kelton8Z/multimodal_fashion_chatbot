var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

$(window).load(function () {
    $messages.mCustomScrollbar();
    setTimeout(function () {
        fakeMessage("Hi there, tell me what look you want. For example, <div class=\"question\">Lexie Liu在timestamped video url of Manta MV那件黑皮衣?</div>");
    }, 100);
});

function updateScrollbar() {
    $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
        scrollInertia: 10,
        timeout: 0
    });
}

function setDate() {
    d = new Date()
    if (m != d.getMinutes()) {
        m = d.getMinutes();
        $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
    }
}

function insertMessage() {
    msg = $('.message-input').val();
    if ($.trim(msg) == '') {
        return false;
    }
    $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
    setDate();
    $('.message-input').val(null);
    updateScrollbar();
    setTimeout(function () {
        jinaMessage(msg);
    }, 1000 + (Math.random() * 20) * 100);
}

$('.message-submit').click(function () {
    insertMessage();
});

$(window).on('keydown', function (e) {
    if (e.which == 13) {
        insertMessage();
        return false;
    }
})


function fakeMessage(msg) {
    if ($('.message-input').val() != '') {
        return false;
    }
    $('<div class="message loading new"><figure class="avatar"><img src="https://api.jina.ai/logo/logo-product/jina-core/logo-only/colored/Product%20logo_Core_Colorful%402x.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
    updateScrollbar();

    setTimeout(function () {
        $('.message.loading').remove();
        $('<div class="message new"><figure class="avatar"><img src="https://api.jina.ai/logo/logo-product/jina-core/logo-only/colored/Product%20logo_Core_Colorful%402x.png" /></figure>' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
        setDate();
        updateScrollbar();
    }, 200);
}


function jinaMessage(question) {
    if ($('.message-input').val() != '') {
        return false;
    }

    $('<div class="message loading new"><figure class="avatar"><img src="https://api.jina.ai/logo/logo-product/jina-core/logo-only/colored/Product%20logo_Core_Colorful%402x.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
    updateScrollbar();

    $.ajax({
        type: "POST",
        url: $('#jina-server-addr').val() + "/search",
        data: JSON.stringify({"data": [question], "top_k": 3}),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
    }).success(function (data, textStatus, jqXHR) {
        console.info(data)
        var top_4_answer = data['data']['docs'][0]['matches'].slice(0,4)
        console.log(top_4_answer)
        var data = top_4_answer.map(match => match.uri);
        $('.message.loading').remove();
        $('<div class="message new">' +
            '<figure class="avatar">' +
            '<img src="https://api.jina.ai/logo/logo-product/jina-core/logo-only/colored/Product%20logo_Core_Colorful%402x.png" /></figure>' +
            '<div class="question">' + data +
            '</div>'+
            '</div>').appendTo($('.mCSB_container')).addClass('new');
        setDate();
        updateScrollbar();
    }).fail(function () {
        setTimeout(function () {
            fakeMessage("Connection failed, did you run <pre>jina hello chatbot</pre> on local? Is your address <pre>" + $('#jina-server-addr').val() + "</pre> ?");
        }, 100);
    });
}

function get_img(){
    // 返回右上角的衬衫
    // 这一行是读取输入框
    var query = $("#message-text").val();
    query = query.replace(/\r\n/g,"");
    query = query.replace(/\n/g,"");
    $.ajax({
         type: 'POST',
         url: $('#jina-server-addr').val() + "/generate",
         data: {'question': query},
         success: function (data) {
             // 这里用于添加回复到对话框中
             var answer = '<div class="item left"><div class="chat-box"><p class="user"></p><p class="angle"></p><div class="message"><p>' + data + '</p><br><a class="send-btn active" onclick="recallQuestion(this);">不满意该回答</ a></div></div></div>';
             setTimeout(function() {
                 $(".list-wrapper").append(answer)
                 document.getElementById("lm").scrollTop = document.getElementById("lm").scrollHeight;
                 }, 500)
             // 这里用于修改图片，通过路由返回的字符串，将对应的图片添加到前端
             var html = '<img src="../static/data/t_shirts/' + data + '" class="render-img">'
             // 选择id为render的组件
             $("#render").empty();
             $("#render").append(html);
             // 清空对话框
             $("#message-text").val('');
         }
   });
}